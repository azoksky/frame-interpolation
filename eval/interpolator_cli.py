import os
import math
import glob
import tensorflow as tf
import mediapy as media
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from absl import app, flags, logging as absl_logging
from eval import interpolator as interpolator_lib, util

# Configure logging.
absl_logging.set_verbosity(absl_logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string('input_pattern', None,
                    'Glob pattern or directory path for input image files. '
                    'If a directory is provided, all *.png, *.jpg, and *.jpeg files will be used.',
                    required=True)
flags.DEFINE_string('model_path', None,
                    'Path to the saved model for interpolation.',
                    required=True)
flags.DEFINE_integer('times_to_interpolate', 1,
                     'Number of recursive interpolations (output frames = 2^times + 1).')
flags.DEFINE_integer('fps', 30, 'Frames per second for the output video.')
flags.DEFINE_string('output_path', 'interpolated.mp4',
                    'Output video file path.')
flags.DEFINE_integer('align', 64, 'Alignment value to pad image dimensions.')
flags.DEFINE_integer('block_height', 1,
                     'Number of patches along image height (1 means no tiling).')
flags.DEFINE_integer('block_width', 1,
                     'Number of patches along image width (1 means no tiling).')

def get_frame_files(input_pattern: str):
    """
    If input_pattern is a directory, fetch all .png, .jpg, and .jpeg files inside.
    Otherwise, assume it is a glob pattern and return matching file paths.
    """
    if os.path.isdir(input_pattern):
        files = []
        for ext in ['png', 'jpg', 'jpeg']:
            files.extend(glob.glob(os.path.join(input_pattern, f"*.{ext}")))
        return sorted(files)
    else:
        return sorted(glob.glob(input_pattern))

def _output_frames(frames: list, out_dir: str):
    """
    Saves the given frames (as numpy arrays) into out_dir.
    Each frame is written as a PNG file named frame_000.png, frame_001.png, etc.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        # Optionally, remove existing files
        for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
            os.remove(f)
    for idx, frame in enumerate(frames):
        filename = os.path.join(out_dir, f"frame_{idx:03d}.png")
        util.write_image(filename, frame)
    absl_logging.info("Interpolated frames saved to: %s", out_dir)

def process_frames(frame_paths, times_to_interpolate, model_path, output_fps=30,
                   align=64, block_height=1, block_width=1, output_frames_dir=None) -> str:
    """
    Interpolates between frames using manual GPU assignment.
    Splits the input frames into segments (with one-frame overlap),
    processes each segment on a separate GPU, then merges the results.
    The individual frames are saved to output_frames_dir,
    and a video is generated as specified by FLAGS.output_path.
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
        if output_frames_dir is not None:
            _output_frames(frames, output_frames_dir)
        media.write_video(FLAGS.output_path, frames, fps=output_fps)
        return FLAGS.output_path

    num_gpus_to_use = min(num_gpus, total_pairs)
    pairs_per_gpu = math.ceil(total_pairs / num_gpus_to_use)
    
    segments = []
    for i in range(num_gpus_to_use):
        start = i * pairs_per_gpu
        end = min(total_pairs, (i + 1) * pairs_per_gpu)
        # Include one extra frame for overlap.
        seg = frame_paths[start:end + 1]
        segments.append(seg)
    
    # Ensure overlap between segments: first frame of each segment equals last frame of previous.
    for j in range(1, len(segments)):
        segments[j][0] = segments[j-1][-1]
    
    absl_logging.info("Splitting %d frame pairs across %d GPUs.", total_pairs, num_gpus_to_use)
    absl_logging.info("Segment sizes: %s", [len(seg) for seg in segments])
    
    # Load one interpolator per GPU.
    interpolators = []
    for gpu_index in range(num_gpus_to_use):
        with tf.device(f'/GPU:{gpu_index}'):
            interpolators.append(interpolator_lib.Interpolator(model_path, align, [block_height, block_width]))
    
    def process_segment(idx: int) -> list:
        try:
            absl_logging.info("GPU %d: Starting segment %d with %d frames.",
                              idx, idx, len(segments[idx]))
            with tf.device(f'/GPU:{idx}'):
                interp = interpolators[idx]
                seg_frames = segments[idx]
                result = list(util.interpolate_recursively_from_files(seg_frames, times_to_interpolate, interp))
            absl_logging.info("GPU %d: Finished segment %d. Generated %d frames.",
                              idx, idx, len(result))
            return result
        except Exception as e:
            absl_logging.error("GPU %d: Exception in segment %d: %s", idx, idx, str(e))
            raise

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

    final_frames = results[0]
    for seg in results[1:]:
        final_frames.extend(seg[1:])  # drop the first frame (overlap)
    
    # Save the individual interpolated frames.
    if output_frames_dir is not None:
        _output_frames(final_frames, output_frames_dir)
    
    media.write_video(FLAGS.output_path, final_frames, fps=output_fps)
    absl_logging.info("Video saved at: %s", FLAGS.output_path)
    return FLAGS.output_path

def main(argv):
    del argv  # Unused.
    # Get frame file list.
    frame_files = get_frame_files(FLAGS.input_pattern)
    if not frame_files:
        raise ValueError("No frame files found for pattern: " + FLAGS.input_pattern)
    absl_logging.info("Found %d frame files.", len(frame_files))
    
    # Determine the directory to save interpolated frames.
    # If the input pattern is a directory, create an "interpolated" subfolder.
    if os.path.isdir(FLAGS.input_pattern):
        output_frames_dir = os.path.join(FLAGS.input_pattern, "interpolated")
    else:
        # Otherwise, use the directory of the glob pattern.
        output_frames_dir = os.path.join(os.path.dirname(FLAGS.input_pattern), "interpolated")
    
    absl_logging.info("Interpolated frames will be saved in: %s", output_frames_dir)
    
    video_path = process_frames(frame_files,
                                times_to_interpolate=FLAGS.times_to_interpolate,
                                model_path=FLAGS.model_path,
                                output_fps=FLAGS.fps,
                                align=FLAGS.align,
                                block_height=FLAGS.block_height,
                                block_width=FLAGS.block_width,
                                output_frames_dir=output_frames_dir)
    print("Video saved at:", video_path)

if __name__ == '__main__':
    app.run(main)
