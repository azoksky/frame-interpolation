import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
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
                     'Number of recursive interpolations (output frames = 2^(times_to_interpolate) * (#pairs) + 1).')
flags.DEFINE_integer('fps', 30, 'Frames per second for the output video.')
flags.DEFINE_string('output_path', 'interpolated.mp4',
                    'Output video file path.')
flags.DEFINE_integer('align', 64, 'Alignment value to pad image dimensions.')
flags.DEFINE_integer('block_height', 1,
                     'Number of patches along image height (1 means no tiling).')
flags.DEFINE_integer('block_width', 1,
                     'Number of patches along image width (1 means no tiling).')

def get_frame_files(input_pattern: str):
    if os.path.isdir(input_pattern):
        files = []
        for ext in ['png', 'jpg', 'jpeg']:
            files.extend(glob.glob(os.path.join(input_pattern, f"*.{ext}")))
        return sorted(files)
    else:
        return sorted(glob.glob(input_pattern))

def _output_frames(frames: list, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
            os.remove(f)
    for idx, frame in enumerate(frames):
        filename = os.path.join(out_dir, f"frame_{idx:03d}.png")
        util.write_image(filename, frame)
    absl_logging.info("Interpolated frames saved to: %s", out_dir)

def process_frames(frame_paths, times_to_interpolate, model_path, output_fps=30,
                   align=64, block_height=1, block_width=1, output_frames_dir=None) -> str:
    n = len(frame_paths)
    if n < 2:
        raise ValueError("Need at least two frames to interpolate.")
    frame_paths = sorted(frame_paths)

    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    if len(physical_gpus) < 2:
        absl_logging.info("Less than 2 GPUs found; using single-GPU mode")
        interpolator = interpolator_lib.Interpolator(model_path, align, [block_height, block_width])
        frames = list(util.interpolate_recursively_from_files(frame_paths, times_to_interpolate, interpolator))
        if output_frames_dir is not None:
            _output_frames(frames, output_frames_dir)
        media.write_video(FLAGS.output_path, frames, fps=output_fps)
        return FLAGS.output_path

    # For 2 GPUs: split the input so that GPU0 gets frames[0:mid] and GPU1 gets frames[mid-1:n].
    mid = math.ceil((n + 1) / 2)  # e.g., for n=6, mid=4.
    absl_logging.info("Total %d frame files split as follows: GPU0: %d frames, GPU1: %d frames (with 1 overlapping frame).",
                      n, mid, n - mid + 1)
    segments = [frame_paths[:mid], frame_paths[mid-1:]]

    # Create one interpolator per GPU.
    interpolators = []
    for gpu_index in range(2):
        with tf.device(f'/GPU:{gpu_index}'):
            interpolators.append(interpolator_lib.Interpolator(model_path, align, [block_height, block_width]))

    def process_segment(idx: int) -> list:
        try:
            absl_logging.info("GPU %d: Starting processing of segment %d with %d frames.", idx, idx, len(segments[idx]))
            with tf.device(f'/GPU:{idx}'):
                interp = interpolators[idx]
                result = list(util.interpolate_recursively_from_files(segments[idx], times_to_interpolate, interp))
            absl_logging.info("GPU %d: Finished segment %d, generated %d frames.", idx, idx, len(result))
            return result
        except Exception as e:
            absl_logging.error("GPU %d: Exception in segment %d: %s", idx, idx, str(e))
            raise

    results = [None, None]
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {executor.submit(process_segment, idx): idx for idx in range(2)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # Merge the results by taking GPU0 output fully and then appending GPU1 output
    # but dropping its first frame (the duplicate at the boundary).
    final_frames = results[0] + results[1][1:]
    absl_logging.info("Total merged frames: %d", len(final_frames))

    if output_frames_dir is not None:
        _output_frames(final_frames, output_frames_dir)
    media.write_video(FLAGS.output_path, final_frames, fps=output_fps)
    absl_logging.info("Video saved at: %s", FLAGS.output_path)
    return FLAGS.output_path

def main(argv):
    del argv
    frame_files = get_frame_files(FLAGS.input_pattern)
    if not frame_files:
        raise ValueError("No frame files found for pattern: " + FLAGS.input_pattern)
    absl_logging.info("Found %d frame files.", len(frame_files))
    
    if os.path.isdir(FLAGS.input_pattern):
        output_frames_dir = os.path.join(FLAGS.input_pattern, "interpolated")
    else:
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

if __name__ == '__main__':
    app.run(main)
