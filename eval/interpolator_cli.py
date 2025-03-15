import os
import math
import glob
import tensorflow as tf
import mediapy as media
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from absl import app, flags, logging as absl_logging
from eval import interpolator as interpolator_lib, util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        # Remove existing frame files if present.
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
    Splits the input frames into two contiguous segments as follows:
      - If total frame count is even, first segment = first half, second segment = second half.
      - If odd, first segment gets one extra frame.
    Each segment is processed on a separate GPU. The individual frames are saved
    to output_frames_dir, and a video is generated as specified by FLAGS.output_path.
    """
    n = len(frame_paths)
    if n < 2:
        raise ValueError("Need at least two frames to interpolate.")
    frame_paths = sorted(frame_paths)
    
    # Enable GPU memory growth.
    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    num_gpus = len(physical_gpus)
    if num_gpus < 2:
        absl_logging.info("Less than 2 GPUs found; using single-GPU mode.")
        interpolator = interpolator_lib.Interpolator(model_path, align, [block_height, block_width])
        frames = list(util.interpolate_recursively_from_files(frame_paths, times_to_interpolate, interpolator))
        if output_frames_dir is not None:
            _output_frames(frames, output_frames_dir)
        media.write_video(FLAGS.output_path, frames, fps=output_fps)
        return FLAGS.output_path

    # For 2 GPUs, split the list into two halves.
    if n % 2 == 0:
        mid = n // 2
    else:
        mid = math.ceil(n / 2)
    segments = [frame_paths[:mid], frame_paths[mid:]]
    absl_logging.info("Total %d frames split into two segments: segment 1 has %d frames, segment 2 has %d frames.",
                      n, len(segments[0]), len(segments[1]))

    # Load one interpolator per GPU.
    interpolators = []
    for gpu_index in range(2):
        with tf.device(f'/GPU:{gpu_index}'):
            interpolators.append(interpolator_lib.Interpolator(model_path, align, [block_height, block_width]))

    # Worker function for each GPU segment.
    def process_segment(idx: int) -> list:
        try:
            absl_logging.info("GPU %d: Starting processing of segment %d with %d frames.",
                              idx, idx, len(segments[idx]))
            with tf.device(f'/GPU:{idx}'):
                interp = interpolators[idx]
                # Process the assigned segment.
                result = list(util.interpolate_recursively_from_files(segments[idx], times_to_interpolate, interp))
            absl_logging.info("GPU %d: Finished segment %d, generated %d interpolated frames.",
                              idx, idx, len(result))
            return result
        except Exception as e:
            absl_logging.error("GPU %d: Exception in segment %d: %s", idx, idx, str(e))
            raise

    # Process the two segments in parallel.
    results = [None, None]
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {executor.submit(process_segment, idx): idx for idx in range(2)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                absl_logging.error("Segment %d generated an exception: %s", idx, exc)
                raise

    # Merge the results by simply concatenating them.
    # (Since we split exactly, no overlapping frames are needed.)
    final_frames = results[0] + results[1]
    
    # Save individual interpolated frames.
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
    print("Video saved at:", video_path)

if __name__ == '__main__':
    app.run(main)
